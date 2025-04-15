Error messages is only grouped by category once and lazy as possible
changed visibility of logger instance on StatsdLoggerImpl class to protected
log dependency resolution process configuration on debug\n\nIn the end this is not relevant for most users which will be happy with the default configuration and do not need to get into the depths of tweaking specific resolutions.
fix comments of BlockWorkerUtils
Remove abbreviated name in getDbConfigUrl\n\nRemove the single use random variable from getDbConfigUrl and inline the creation
OGM-<I> Relax regexp used to detect primary key constraint violation\n\nMore modern versions of MongoDB have a different syntax.\n\nThe regexp we use now should be sufficiently future proof.
Fixed failed test after not having the initial url loaded directly
[Java] Reuse local variable rather than do repeated lookups.
Minor bug fix for fromStringList
[FLINK-<I>][state/changelog] Don't discard private state\n\nDiscarding private state for the changelog backend\nwill be addressed in FLINK-<I>.\n\nDiscarding it now fails some tests by invalidating\ncheckpoints on abortion.
Added GLM1_RHO as the default temporarily.
Fixed order of configuration and cleaning in Spring Security context holder.
adding constructor that receives both name and weight. closes #<I>
Fixed private method in luttice test
Remove OutputSymbol from NodeRepresentation
DataPager added reload() method
Temporarily remove testcase from checkpoint FAT.
[RHPAM-2] NullPointerException during transaction rollback when using CMT (#<I>)\n\nif it is not the transaction owner we try to rely on the transaction registry\n\nand set the current tx associated as a rollback
Removing change to BasicDefaultIgnoreExceptionsTest
HTTP: fixed bug on returning response when accepts is */* like with curl
SearchDialogAction: Added labels, looks nicer. But still some layout\nissues.
[Java] Remove outdated comment from CapabilitiesDirectoryImpl
Add line breaks to test failure message.\n\nTrying to improve readability. Message is now:\nExpected:\nsyntax....\nGot:\nsyntax...\n\n[Issue: ]
Rewrite WaitStrategyTestUtil in spirit of JUnit 5
Reformatted to cut down Checkstyle issues.  Noticed that Lombok source is not part of Checkstyle report, but I think it is due to <URL>
Bug fix: bias feature was still being added in SentFeatureExtractor.
[WFLY-<I>] Update core model version
skip nested city objects when unmarshalling CityJSON
better ip address for client side.
zuul-core: make client connection shutdown depend on the channel\n\nRather than wait for a request to come in to finish off a channel, just wait on the channel closeFuture instead.
That ended up being called PresentsTestBase.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/narya/trunk@<I> <I>f4-<I>e9-<I>-aa3c-eee0fc<I>fb1
[Java] Append single char rather than a string.
Add method .verbose(boolean verbosity)
Reduces log level to avoid logs flooded with one log entry per imported record.
XCOMMONS-<I>: Move from Class to Type as Component role\nFix bug in previous commit
resolves issue #<I> by closing containerless container
changes due to changes in JP <I> API
Added extension init/finish calls
Added some extra tests for array creations
Update Postgres Test Container initialization to address removal of default ctor in Test Container update.
Fix typo - Tentative fix for issue #<I>
Using month-date format "MM" instead of "MMM" since eg. month "Jun" doesn\'t work on russian VM\'s
models: ParseVariant checks if ref or alt is "-" and saves an empty string.
Use different someuser for tests from ko, crud and visual projects
Clean up imports on the feature demo.
Do not remove quotes from identifiers (e.g. sequences) when used in sequenceNextValueFunction.
synchronized validation to avoid concurrency issues like FWK<I>
Added byte as a supported type
revises leader election example for clarification
Fixed a bug in the recycler
Temporarily removed test with cyrillic that works unstable in Chrome.
Use loader's classloader for resource access.
Fix javadox in XMPPConnection
Wrap at <I> columns.
mongodb: Added "skipCount" options. Avoid count "numTotalResults"
Escape escape characters which appear on Windows paths.\n\nThe current test setup does not afford an easy way to test this.
Add started() method to indicate if REPL is started
Add test: typing a character should move caret after inserted character
changed bind() to rebind() so that it doesn't through during multiple\nuser logins. JNDI manager should be fixed to handle multiple logins without\ndoing all this extra work.\n\n\ngit-svn-id: <URL>
Do not call serverTracer.clearCurrentSpan() at the start of a span
Reduce runtime for DLProstateTest.
Exposing ClientFactory so users create Senders and Receivers.
Set the fps counter target framerate to <I> rather than to the composition framerate\n\nAndroid's target framerate is always <I>
Fixed NPE in processing null-returning attachment methods (fixes #<I>) (#<I>)
Refined OpenJDK specific test.
Excluded nobs from early stopping.
Enhancement: Adds lookup for relationship schema information.
Remove unnecessary 'SuppressWarnings' annotation.
fix bug with end of tick 0 not properly emitted sometimes
The client needs to be closed after the deployment.
Do not create 'tomcat<I>' directory under project root directory\n\nMotivation:\n\nUnmanagedTomcatServiceTest does not specify the base directory of the\nTomcat, which makes the Tomcat create an empty 'tomcat<I>' directory\nunder the project root directory. It could be created somewhere under\n'/target' instead for less distraction.\n\nModifications:\n\n- Set the baseDir of Tomcat explicitly\n\nResult:\n\nCleaner build
Condition is the wrong way around: We want to check that we have service name but no version
added to DeepDive<I> w/Samantha
Avoid creating the empty object for S3 root\n\n### What changes are proposed in this pull request?\nAvoid creating the empty object for root\n### Why are the changes needed?\nTo avoid some customized S3 object storage error.\n### Does this PR introduce any user facing changes?\n\npr-link: Alluxio/alluxio#<I>\nchange-id: cid-be7d<I>d4dc9a<I>a<I>d0e<I>c3b4ea1a<I>e<I>
set visitSpreadFragments to true
Fix server start failure caused by version negotiation
fix NullPointerException in case table "sms" is empty (github issue#3)
Fix a typo that prevented the right math for deciding whether to cache images or not.
Don't read parameter. Input stream of request beccomes 0 bytes
don't start sparkUI in unit test
for real device, leave some time ( 1 sec ) for the stop command to be processed, so that instruments terminates correctly.
JavaFormatterTest expects SHA value which is not the same\n\nNot sure if this is a defect or not.  This issue is preventing local\nbuild and travis ci build.  It did not exist until after changes in\npoms.  Possibly due to newer version or something but not sure how this\nworks or is expected to work.
fix javadoc at line <I>.
Simplify boolean logic as Martin suggested
The highly important commit of changing a log line
Fix adding more than one percentile aggregation\n\nPrior to this change, the user was unable to create more than one\npercentile metric for same field in aggregation. Since the aggregation\nseries name was matching on aggregation name and field.\n\nThis change will add the percentile to the aggregation name. That way\nthe user can add more then one percentile aggergation.\n\nFixes: #<I>
remove TestRule impl pending <I> (will get it out of SCM when we need it.
Removed commented lombok annotation @Slf4j
[en] move rule so comment ("specific to English") makes more sense
Changes for setBoolean to INT<I>
Improved XHTML compatibility.
Set the source info of a transpiled class member correctly\n\n-------------\nCreated by MOE: <URL>
Fixes continuous sound issue with api >= <I>
- webapp: making sure the list is returned no matter what comes\n\ngit-svn-id: <URL>
#<I>: set protocol version to <I> (#<I>)
Fix principal-match-report (missing elements)\n\nAdd recipient to freebusy response
Fixed crash when Steam is running, but no steam_appid.txt found for debugging.
include fix for connections in Renjin
[JBPM-<I>] ProcessServiceImpl.getAvailableSignals should obtain ProcessInstance in read-only mode to avoid race condition
Added support for KUBECONFIG environment variable with multiple paths.\nUse first kubeconfig file.
Detection of SMILES files was dropped as it is too difficult; but the test was not removed yet\n\n\ngit-svn-id: <URL>
Allow Conscrypt to be used on Android (#<I>)\n\n**Motivation:**\n\n\n\nFollowing up on <URL>. But the current availability check prevents Netty from using it (the call `PlatformDependent.javaVersion()` returns `6` on Android).\n\n\n\n**Modification:**\n\n\n\nCheck whether the java version is above 8, or if it's running on an Android device, to flag Conscrypt as available (besides the remaining checks).\n\n\n\n**Result:**\n\n\n\nNetty with TLS runs fine on Android 👍
don't clear corpus selection when adding node in query builder
applying very uncommon 2-space indentation
#<I> moved file related descriptors to jqassistant.plugin.common
Fixed recursive initialization bug.
removed unused import from post request action
minor refactor: an additional utility method,
resolved #<I>, needs testing on fred's project :-)
Removed the override annotation, because by default we use hdfs <I>
Remove premature optimization\n\nRemove code that avoids extra allocations but that could lead to an\nexception if bytesNeeded > hard-coded array length\n\nJAVA-<I>
tests: make sure that the default manager exists always
Workaround for NPE on ExecutorService.shutdownNow()
MODE-<I> Improved how test files are accessed
ENG-<I>: Fixing compile error missed from merging from master.
Change log4j root logger level to INFO.
Add misssing connection id
removed empty line between JavaDoc and method
Add test case for selecting nullable reference
Renamed method to 'getCachingAndUpdatingParser'
Updating tax id field in tests
Update AccessLevel.java\n\nOn gitlab.com I got \n\n```\n\norg.gitlab4j.api.models.AccessLevel forValue\n\nWARNING: [5] is not a valid GitLab access level.\n\n```\n\nso I've added that enum here
[xtend][quickfix] Quick fix for missing abstract method implementations
Disable anonymous auth for resource server
#<I> unify scanning of container items like directories and archives
fixing package name (#<I>)
Assert temp file ok in test setup
System.out.println removed
fix: demuxl and muxl tests
PURE-<I>: Only the opacity of the html container that represents the given RasterLayer is changed.
Check that docker-machine is not present properly
making Connectivity class more hermetic
added handling of hd.connectUri() == null return;
Prevent ServerInfo from initializing too early
Converted into more stable, recommended use of place definition\n\n\ngit-svn-id: <URL>
FIXED #<I> - Thread-waitForPendingFutures is leaking\n\nFrom thread-dump and code review, it seems future is already DONE or CANCELED state when calling `Future.get()`, so it seems notification is not fired. As solution, instead of using `Future.get()`, use `Future.get(timeout, unit)` with checking `isDone()` and `isCanceled()`
Added explicit cast in unit test to help Java <I> with type inference that Java <I> handles
Fixes exception message typo in mockmaker (#<I>)
JBPM-<I> - Make commands that contain a "process instance id" field implement the ProcessInstanceIdCommand interface
[Java] Increment of invalid packets counter needs to be concurrent because it is used by both sender and receiver threads.
Fix typo in exception message: "row" => "column"
SONAR-<I> Fix missing line with Git
Made Serializable as test fix
clarify usage of ConcurrentHashMap pushIfAbsent
Avoid creating an iterator for every node when traversing the AST during check\nconformance.\n-------------\nCreated by MOE: <URL>
Fix Enum allowable list does not display in the Swagger Model
Add check to testJavaScoring that MOJO is actually Serializable
Support remote execution in distributed members group.
Update SSH2HostbasedConnect.java
Swing: fix back/forward with details
Using final instead of abstract for static utility classes\n\nNetBeans <I> is warning "Constructor is never used" when abstract, and\nthis cannot be suppressed with `@SuppressWarnings("unused")`.
And updated the call from one to the other...
Fixed typo in the browseFrom method name. Fix #<I>
force eagerness of processing of SegmentMetadata queries on the processing executor by converting the Sequence into List
Fixed cloning of properties with null values by always using HashMap (fixes #<I>)
Fix typo in dialog box for nonlocal database
Bugfix in glm scoring.
Ignored test that causes problems.
GH-<I>: polishing user property level override to allow command-line override even if seed is specified.
reset chart image name inside report if an error occured
Fix 'oops' in detecting correct collection type.
This annotation isn't useful here.
Added assertFloatsEqualDefaultTol to testing utils.
Handle case where class' Path annotation contains a trailing slash
Fix time unit in process.start.time and process.uptime
Use delta for float comparison in ResizeSettings
Check if we have a memory cache
throwing exception on getting / setting a header with id <= 0
Fixed crash when classpath is a directory. (#<I>)
[ALLUXIO-<I>] Fix unit for master client thread pool keep-alive time (#<I>)\n\n* Fix unit\n\n\n\n* Set unit explicitly
- fixed ExampleVideoMosaic\n  * slight change to decoding with Xuggler was messing it up\n  * just added more RANSAC iterations
Add reference to EOM issue from unbounded body size
Fix classpath splitting on Windows\n\nWe delimit classpaths in the properties file inside our jar files with\na colon. But, on Windows, a colon is also used to separate drive\nletters from the rest of the path. So, we detect the case where we\nsplit a path up into its drive letter and path and reassemble that\ninto the correct path.\n\nWe can't just rely on File.pathSeparator because then the built Jar\nwould only work on the same OS Family (Windows vs Linux/Mac) that it\nwas created on.
Sim date is no longer showed on action preview
Make constant for <I> status
Reduce AddonRepositoryImpl logging level to FINE
Added method clear one view from the cache
[Java] Increase test time to <I>s.
When flushing transaction log, ensure that the writes are done in local mode!
A jab to the left, a jab to the right, next thing you know you're boxing.\n\n\ngit-svn-id: <URL>
Shows more info about throwable\n\nShows more Android specific info about throwable
[en] decrease prio of a generic rule
Updating OperaDriver class description
data type from stn to radial
Improve descriptions for HamcrestConditions
Removed auto-reload of functions at db.open().
Make test case easier to debug by using two different type annotations.
Fix race condition by not caching AstNodes when using eager execution
PERFORMANCE:\nonce result becomes NaN, it will be always NaN on adding anything
Add permissions for the cluster config\n\n(cherry picked from commit <I>c5b5b<I>dd0fbd<I>ea<I>b<I>cbbe5ed<I>dc<I>)
Fixed a goof where we were checking for a thing that would ALWAYS be null.
use ConcurrentHashMap to avoid concurrency issues
smaller variance in gaussian prior in test leads to better accuracy
UNDERTOW-<I> Reverse proxy does not work for POST requests when HTTP/2 is in use on the front end
Get default watermark from BaseSource when there are failed or aborted previous states without committed states\n\nRB=<I>\nR=stakiar,kgoodhop,lqiao,ynli\nA=kgoodhop
AS7-<I> correct recalculation of nextExpiration after restart
<I>: Add descriptions of Java SE 7 and 8 language changes to SourceVersion\nReviewed-by: jjg
Added tearDown method which resets the context to null to prevent occasional breaking of other test classes.
Fix map access with wrong key generic type
Change generic signature of Validator.checkRange method and Validator.CheckRange class.
[BAQE-<I>] Stabilize flaky test: AsyncMultisubprocessTest (#<I>)
Remove the usage of the deprecated Metrics from reactor-core (#<I>)\n\nThis is additional change to #<I>
Fixing dbCreate() to run\n\nI think we need run here.
[PAXWEB-<I>] - pax web still using uninstalled bundle
Avoid NPE when underlying result point is null, for some reason
test fix for japanese last nengo\n\nreiwa is now open-ended, but not heisei
Migrated to using cached thread pool for async calls to increase transaction throughput, as per <URL>
Part of FCREPO-<I>. When the policyFinding fails, log the error on the warn level, as the stacktrace is thrown away afterwards. Should hopefully make debugging easier
Reduced the threshold for TraversalStrategyPerformanceTest passing again CTR\n\nThis is still failing randomly - nost notably in Travis.
Don't trust stored total when determining whether or not to expand.
fix for issue <I>: Average number of requests per minutes seems to be wrong
Fixed missing return on erro in GameUpdateTask
note some todos in the PoolTest.
Activiti Forms with interdependent Properties\n\nBe able to get the Vaadin Form in Activiti Explorer from any Renderer.\n\nThis enables to program widget interdependency (cascade combo boxes for example)
Remove ICU4J from SdkConfig, there is no need to load this jar through the instrumenting classloader.\nThis can be provided like a regular dependency which allows applications that use ICU4J themselves to\navoid version skew by Robolectric specifying a specific version.
Fix memory leak due to connectivity manager keeping activity context.
Prevent out of bounds exception.
s/new new/new/ (#<I>)
Fixed issue <I> about specify a field as TRANSIENT at document level
added missing final to xml extractor helper
added methods back that are removed by mistake
improvement getNodesValues(List<String> nodes, int... excludedColumns) method in Tree
Issue #<I> - Removed unused
filed a bug as suggested by Jesse\n\ngit-svn-id: <URL>
Handle value flag also for getText()
Introduce verificationSuccessful boolean
Fixed IMap replace() always generating ADDED events.\nThis should only ever generate UPDATED events if successful.\nThis seems to have regressed in Hazelcast <I>.\nHazelcast 2.x seems to have only generated UPDATED events, see <URL>
Test stability: InteractiveTestJob default wait 5s-><I>s (#<I>)
made the documentation of the timesTransposed(vector, matrix) more precise
[Java] Slightly more efficient contains operation on URI.
XWIKI-<I>: Add support for nested spaces in XAR export/import/format\n* fix mistake in previous commit
ensure projects not already on user's machine are installed
refactored public getrelations methods in DE9IM
[Java] If recording fails to stop, restart the Election.
Added an example on how to change the background color for "columns" inside the ResourceCalendarView.
Fixing issue where date picker was not closed when closing parent\ndialog.
[xbase][linking] Avoid infinite recursion for broken cross references
Make sure ping thread runs and set all servers alive before running junit.
Don't use an immutable list here.
Use tempDir property rather than repeating the call to\nSystem.getProperty().
JDK-based logging configuration is sophisticated enough
Issue #<I> - missing field added to Photo type
skip instead of throw error on non-linux os
Added waiting utils with timeout control in test commons.
* fix selection not set for profile items if the accountList is currently visible\nFIX #<I>
Further work on TORQUE-<I>, no longer losing query-string parameters.
fix direct login /cas unauthorized service error with MFA trust… (#<I>)\n\n* fix direct login /cas unauthorized service error\n\n\n\n* fix direct login /cas unauthorized service error
Minor logging fix in TimeScheduler3
#<I> Add support for event notifications\n\nFix test to use all types of event listeners
query's getDeclarations needs a null or empty check
small code cleanup, thanks to Rémy Léone
[SHRINKWRAP-<I>] Do the best we can here to let the user know why they are deadlocking
[FLINK-<I>] fail if user jar contains no executions\n\nThis closes #<I>.
Graduating Stackdriver integration from Incubating status, fixes gh-<I>
MODE-<I>: Reset path when workspace or repository has been changed
handle double calls to getRequestContent
fixed sonar issues (removed unneeded method)
Added multiple global error handlers / writers
Fix bug in ClassFileWriter IINC handling.
Updated LB capabilities to match core
OGM-<I> Remove warning and avoid getting the dialect in integretion test
Correction on StringBuilder default init size\n\nMotivation:\nThe default StringBuilder size is too small (data.length + 4) while it will be 2*data.length (byte to Hex) + 5 "-" char (since 5 peaces appended).\n\nModification:\nChanging initial size to the correct one\n\nResult:\nAllocation of the correct final size from the beginning for this StringBuilder.
Add missing return statement to switch case in TypeService
Travis seems not to like connections to localhost.
use getTokenizingCharacters() instead of the constant, so sub classes can overwrite just that method
Grant access to file converters for other modules\n\nFixes #<I>
Avoid over-allocating a buffer (#<I> and #<I>)
fix: remove public method incautiously\n\nhandle redirect on DownloadConnection layer
Remove trailing whitespace from LiteBlockingWaitStrategy.\n\nThis was causing checkstyle to complain.
Trigger full refresh after resizes
For an unknown reason a different exception is raised in NioWorker.read()
Attempt to prevent race conditions. Wild guess.
Fix flakiness in `CollectorContextTest` (#<I>)\n\n`testCollectorContextWithKeyword` was failing under [NonDex](<URL>) due to implementation-defined ordering of the list of context values.  This PR fixes that flakiness by sorting the list in natural order before inspecting it.
Fix: tokenizer of Hex values may need to ignore unexpected white space \n\nThis has showed up in InlineImages, which do not appear to work for\nother reasons.
Add isInputMethodTarget() to AbstractTextViewAssert
added `final` to the members of `Platform`.
Prevent crash on ranging by serializing ranging callback for ScanJobs
Remove non-necessary trown exception
JBPM-<I>: When delegating a task, afterTaskExited is also triggered - change of the status was set after invoking listeners
Fix for multiple ":" in string (e.g. timestamp).
fixes Locale problem when getting valueof AreaFinder (#<I>)
GUACAMOLE-<I>: Do not wait forever for streams to be notified.
Fix elapsed time display in logs
Constructor should still be private - debugging from last commit.
ISPN-<I> Single-server shutdown authz permissions
Fix EdgeTest (include missing import)
ExpressionEvaluator now tolerates equalities between XSD.STRING and the SPARQL_LANG function.
[RHD] Enabled parallel segmentation on the variant graph (acyclic), temp. disabled cyclic graph.
Fixing compile errors on the PR
Add separator and quote option in CSVExporter
simplify reverse using flatMapIterable as suggested by @akarnokd
Remove override of unused method we're going to delete from wildfly-core
#<I> Unable to set expires attribute in Cookie (#<I>)
ORC union type is not supported and should throw
Added hint for findbugs.
lets default to not creating Route / Ingress resources as we now use the exposecontroller to decouple that logic
record when the server is initialized so that it tears down everything properly.
[Hitbtc] OrderId is taken form request (if not null) instead of generating from nonce
changed getTxnControler from private to protected
Add better docs to the WebSocket example.
COMMERCE-<I> We shouldn't be searching for addresses under a groupId outside of site scope
SimonStewart: Make the IE driver consistent with the chrome and firefox drivers\n\nr<I>
Renamed to JOBLOG_FILENAME
Corrected model set for jade rendering\n\nJadeModel implements scopes for the model data. Earlier only the current\nmodel was being passed which caused missing of config model. So changed\nit to pass current instance of JadeModel which has get method\nimplemented to iterate through all scoped models. Earlier implementation\nwas causing model.get("config") to be null in TagExtractor for\nJadeEngine but was working properly for other engines.
Disable automatic checking for ajax attributes during repeatUntil call
[Java] Spy source identity should be IPC and not the publication destination channel.
Avoid array allocation by using #charAt instead of #toCharArray
Remove unused `javax.annotation` import which breaks Java <I>+
Added event listener registration in application context
[FIXED JENKINS-<I>] Reverting getDownloadableId from #<I>. Not compatible.\n\n(cherry picked from commit <I>db2a<I>fc<I>b8fb<I>a8d<I>cf<I>b<I>c6e2)
Fixes issue <I>
Allow initializing the FontIconTypefaceHolder with File
Deal with null charges on 'this' too\n\nChange-Id: I<I>f5cea8fef<I>c<I>bd<I>f<I>d<I>a<I>
Simplify the EVENTS initialization
Add @since TODO on new method
- added fast variant of qr code detector config
If we don't find any classes to analyze, don't augment the auxliary class path\n\n\ngit-svn-id: <URL>
ups do not trim time annotation too much
fixing jndi name of process archive service
CAS-<I>\n\nfixed potential null pointer exception in ServiceThemeResolver.  Thanks to Cyrus for reporting/patch.
-Interrupt execution if a robot dies due to old age. Otherwise it\ncontinues executing
CB-<I> android: Added logic for searching sensors from Samsung vendor
Switched from add to put, forcing a wait until the queue has space.
Get converts to booleans
Set tsd.query.allow_simultaneous_duplicates = true by default.\nThis was causing some conflicts and confusion so we'll let folks\nenable it if they have problems with abuse.
HAL-<I>, JBEAP-<I>: make keepalive-time/time required
[Java] Add missing tag to dissector.
Making InputService more resilient against server failures
[JENKINS-<I>] added comments to QueueListener
add check to SimpleConfigList that resolveStatus is correct
moved testExecutionShouldNotTimeoutIfTraceable_WithMultiCallback to SlowTest
Don't reclose an already closed connection, it generates a warning.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/narya/trunk@<I> <I>f4-<I>e9-<I>-aa3c-eee0fc<I>fb1
Added the code to move the output of tests into the `target` folder into LocalTestEnvironment.java
[Java] Start hashCode at 1.
Enabled AAD for conditional expectation.\n\n(This was disabled for testing purposes, e.g. to generate the data in\n<URL>).
Use getFailOnError when Content-type does not start with image\n\nThis case can happen with Qgiserver, when for exemple the layer name is\nwrong.\nQgisServer will respond (Http <I>) whith an XML containig the error\ndescription.
Reduced number of path to speed up test.
use seat version 3 instead of 4, as we don't really need 4.
Don't run System.exit() in test case.
Garena incorrect chatUrls\n\nBoth Vietnam and Indonesia are garena owned servers.
Extend reception bean\n\nAdd a new method to retrieve messages already received by the bean\nwithout waiting.
Add log to all kind of loadMetadata
Make loading the prepared scan result a bit more robust
refactor, fixes, more test
Update version string to <I>.
another tweak to the same test to reset the saturated object after each iteration, otherwise reflexivity is really checked only on the first iteration
Buffer position is updated during data write.
JBRULES-<I> Explicit 'and' is not working\n-It wasn't removing direct redundant ands, now  fixed.\n\ngit-svn-id: <URL>
implement getVersion and isVersioned to satisfy the new methods defined in the PersistentEntity interface.  These methods are currently throwing UnsupportedOperationException
avoid signalling empty batch to BatchStartAware handler
Skip Conscrypt test CA bit (#<I>)
Don't warn about open streams not being closed in main methods\n\n\ngit-svn-id: <URL>
Increase queue capacity.\n\nI am skeptical about making the queue size configurable because this\nwill add more complexity to configuration. Maybe a good default value is\nsufficient for today's hardware. May change my mind later.
fix toString() so that you can actually select the external rules just loaded (could be reproduced with default locale set to English)
fixing the 'unknown.host' test in ClientRequestExecutorPoolTest\n\nit turns out that somebody has registered 'unknown.host':\n\n    $ host unknown.host\n    unknown.host has address <I>\n    unknown.host mail is handled by <I> serpens.uberspace.de.\n\nwhich is kinda of annoying. Have switched to using unknown.invalid as\nthis is reserved and should never resolve.
Check if ErrorStream is null before trying to read it.
Make OnExpressionCondition package private\n\n(consistent with the other Condition implementations)
Removing outdated references to Exceptions#bubble(Throwable) (#<I>)
Reorder assignments to match order of parameters.
fixed HALBuilderTest with new grakn syntax
Add support for multiple type for PlacesApi
Close WebSocketServer's selector to free socket (#<I>)\n\nclose the selector when the WebSocketServer's selectorthread is\n\nfinishing. This allows the port to be freed on stop() and another\n\ninstance to bind on the same port
MInor: added a check in ODocument to avoid wrong settings of multiple parameters
- delete must be made without checking delete access\n\ngit-svn-id: <URL>
Remove feature to click on any occurrence of 'value' if 'place' (i.e. 'select' element) cannot be found, see #<I>
perf: cache result of parsing server_version\n\nSave version number after parsing it from string.\nNo need to recalculate it again.\n\nBackend does not always send server_version_num, thus the driver falls back to string parsing.\nSee discussion on hackers list: <URL> which ckecks for server version.\n\ncloses #<I>
Fix NPE in nodetool when gossip isn't initialized.\nPatch by brandonwilliams, reviewed by jbellis for CASSANDRA-<I>\n\ngit-svn-id: <URL>
Hide obvious beginning of path in UI view
lol rm an intentional debugging break
Fix for multiple AllDay events on one day
Add PostmarkResponse.toString\n\ncopying the pattern from PostmarkMessage
PAXWEB-<I> Correct double negation to allow reassocation to the default shared http context.
now making sure not to return Calcium's when Calpha atoms are requested\n\ngit-svn-id: <URL>
refactor: Use multi-catch for exceptions in ConnectionFactoryImpl
can now parse exponential notation
Minor: profiler dumps the configuration when started
Reduce memory consumed by flat map value block
Deprecated DBCollection.apply and DBCollection.doapply methods, as they will be removed in the next major release.\n\n JAVA-<I>
Fix <I> broken test (#<I>)\n\n* added a paging api to ijobcoordinator\n\n\n\n* adding changelog\n\n\n\n* moved location\n\n\n\n* added a comment\n\n\n\n* Added Sort parameter to JobInstanceFetchRequest\n\n\n\n* Bump HAPI-FHIR version\n\n\n\n* Updated JPA api to include sort\n\n\n\n* Add changelog\n\n\n\n* Fix test
Exception handling was not implemented as intended, fixed.
renamed method addItem to reAddItem
always check closed to avoid stray loader exceptions
cli: when node name is complete and suggesting the next separator, return the buffer length as the index instead of the node name start index
PeerGroup: make selectDownloadPeer non static so users can override it and control selection behaviour.
Don't reset last voted for one heartbeat timer resets.
Update 'getEC2Provider'
Use repository executor when processing watch request (#<I>)
Timeout for the rollback test.
Did some method renaming to make more sense.  Also added comments to places that look incorrect.  Sigh.
catch all errors (#<I>)
Add XLNet to zoo model ignores (#<I>)\n\n* ignore xlnet
Fix mistake in documentation. Re-format documentation.
BlockFileLoader: On an unexpected problem, wrap exception with additional information.
Changed driver version to <I>
[ALLUXIO-<I>]
Return unsent messages from the retry queue, too.
Adds missing getters in ResponseWrapper
turns off generation of JPA annotation for presentation tier project nature
Removed Generics - apparentally we compile with source <I>
added handling PUT on servlet level
Now suppresses (but logs) error on rollback, which seems to be masking a real exception
SHRINKWRAP-<I> Change to substring based on (name - .class length) instead of regexp replace
Correct testReusingBuilderObjectWithMultipleThreads() and make the test code run in parallel. This could affect #<I>
fix for FB <I>
CE-<I> better handling of Close()
WYRL: some progress ... things are looking a bit better, although it's not working perfectly yet.
Remove unused field from CompilerConfig
Update RuntimeConstants.java
Additional Closeables for 'CloseableValue::trying' method.
Prefill wizard dialog with existing param String - accept any order for metric place
#<I> Fix to make javadoc happy again...
Fix null pointer exception in ApiTest.testAttachments()
[LO extension] solves issue #<I>
Add test that PathService ignores empty strings.
ExportManager.getAllNames() do not return names with equals anymore
SEC-<I>: Make default SavedRequest a "GET" in test to prevent NPE.
GitHuh Issue #<I>: Fix stupid signature typo
[hotfix] Remove exception stack trace from Emitter shutdown message\n\nBefore, we were printing the exception (and stack trace) when the Async\nI/O Emitter was receving an interrupted exception. The interrupt,\nhowever is part of the normal shutdown process of the Emitter and the\nlog message was causing concern because a stack trace usually indicates\nsomething went wrong.
[HWKINVENT-<I>] Whoops, fix the compilation failure.
Removed @SupressWarnings in ViewFetcher.java
Use a Deque instead of Stack.\n\njava.util.Stack is synchronized which is not necessary here.
Fix WebSecurityConfigurerAdapter Javadoc\n\nThe constructor's Javadoc was incorrect. This commit\nfixes it.
[Java] Remove redundant method.
Convenient creation of temporary files in the test commons.
dao methods now can't throw SQLException
JBRULES-<I>: Code completion in DSL in when-part does not work properly anymore\n - fixed so that code completion in LHS also works after first character\n - also fixed an issue where a DSL mapping file was not written correctly on save\n\ngit-svn-id: <URL>
signal server error if roles could not be loaded
DROOLS-<I> - kie-server-tests: extended processing time for JMS job executor\n\n * closes #<I>
FIX: add onload handler to rewriting of js
Fix again the stoichiometry order reverse List
[Java] Make synchronisation around shutdown latches more robust, add a method to signal all latches
Clear JAVA_OPTS in CLI tests; avoid permsize config error with Java 8\n\nWhen a Java 8 JVM is launched with -XX:MaxPermSize a warning message\nis output indicating that the option will be ignored. This causes the\nCLI tests that assert that no error output has been produced to fail.\n\nThis commit updates the CLI's integration test harness to remove\nJAVA_OPTS from the environment of the CLI process. This prevents any\nunwanted max perm size configuration from leaking into that\nenvironment and breaking the build.
RoundRobinExecuteWithFailover: int overflow fix.
Add missing @JsonCreator annotation to GrokPattern
Fixes #<I>\n\nRe-worded warning log message.
WFLY-<I>  support any loopback address via the -b argument, without xml files changes\n\nwas: <I>a7ff<I>fa<I>e7c<I>fbe1d<I>ae5a<I>
Add more info in a preconditions check\n\n-------------\nCreated by MOE: <URL>
Defining APPLICATION Priority for CustomPathResolver
Configure Build Driver to use internal network.
adding supresswarning annotation for unchecked code
Do not log other kinds of Exec typo's
warn if topic not found (#<I>)
Sanity check on running patch scripts for eclipse - if its already been done we do not run them again.
MODE-<I>  updated javadocs to indicate support of configuration being an empty string
PeerGroup: print a log warn if there's no blockchain.
[SMALLFIX] Fix web UI fail to load issue (#<I>)\n\n* fix web UI fail to load issue\n\n\n\n* catch error from ufs creation\n\n\n\n* bail out to avoid NPE
JimEvans: Fixing find by error from previous commit.\n\nr<I>
Removed LOG from IndexedSet for jira TACHYON-<I>
AlexisVuillemin: Fixes for IE the java interface contract about error handling for FindElement. The only exception the collection of FindElement{,s} can throw is NoSuchElementException. This submission ensures no other RunTimeException can be thrown.\n\nr<I>
ENG-<I>: Fix bad prev commit.
Added text to @return tag
change dns cache file create logic
Removing access to postEffectManager since its not longer needed
Fixed server-side JavaScript scripting without breaking the Neo4jService..
DECIMAL -> DECIMAL added to JdbcTypeMapper.
Remove override stopUFS in LocalTachyonCluster
[#<I>] Fix typo in DefaultStompFrame.toString() method.\n\nMotivation:\n\nDefaultStompFrame.toString() implementations returned a String that contained DefaultFullStompFrame.\n\nModifications:\n\nReplace DefaultFullStompFrame with DefaultStompFrame.\n\nResult:\n\nLess confusing and more correct return value of toString()
Hack to reduce warning printouts until Bethe free energy bug is fixed.
Task #<I> - Make sure that all jar manifest information is available
Add title and snippet to map marker.
Fixed batchSynonyms targeting slaves (#<I>)\n\nThis is a `build` call, we need to use the build hosts.
[enhancement] added equals and hashcode
Replace constants (static final CONSTANT_CASE) declaration type which use the general collection interface (e.g. List) with an immutable type (e.g. ImmutableList).\n\nFor constant field declarations, you should use the immutable type (such as ImmutableList) instead of the general collection interface type (such as List). This communicates to your callers important semantic guarantees.\n\n-------------\nCreated by MOE: <URL>
Added a "(none)" quote char to CSV datastores
Consistent tag name between Java and .NET libraries\n\nFavouring the more descriptive form.
deprecate Poller constructors that create pollers not affiliated with a context
SEC-<I>: Forgot to commit tests.
Fix test flakiness. Returned cells might not be sorted by timestamp.
Modified GroupProperty to return an empty object instead of null for grouped properties. Work in progress on this.
Refactored to support new Completes<T> nested async actions.
Logging to help debug repeated subscription to cluster objects.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/narya/trunk@<I> <I>f4-<I>e9-<I>-aa3c-eee0fc<I>fb1
Fixed issue on server-size tx.commit(): the cache wasn't updated.
[hotfix] Remove unused imports from MetricOptions
[Java] Default archive low storage space threshold to default segment file length.
Fix bug in AsyncTcpSocketImpl with multiple write calls
use <I> as default servlet version
AlexeiBarantsev: Fixing broken test canStartMultipleIeDriverInstances, and adding a block to quit IE even if the test fails.\n\nr<I>
made unidrectional back reference optional so entities can be saved without having to add them to the parent collection. Fixes GRAILS-<I>\n\ngit-svn-id: <URL>
Improved locale handling for resource info box.
add 'ô' character to allowed chars in WordListValidatorTest
Added default groupId to FractionList
min of <I>L for multipart
Issue #<I> Fix OffHeapValueHolder.writeBack() - writeContext needs to be flushed
Closes Issue #<I> : don't respond to (invalid?) UPC-E codes that aren't 8 digits
changed BoxedUnit.toString to {} from ()
support asynchronous tests in browser mode
Add missing getters/setter\n\nfixes gh-<I>
Fix tests after #<I> (underscore removal)
Add unit tests for SRID constraints
Fix invalidateDrawable method behavior for non-Lottie images (#<I>)
Fix the missing initialization of Block Array in BlockLockManager
Improve Javadoc spacing in FileInStream
Remove only artifacts that are not connected to any BR.
ISPN-<I> Remember to shutdown the periodic flush executor.
[Service Chaining] Copy shared worker pool with from incoming cMsg to response cMsg for Service chaining
BI-<I> - Calling the build start time method that accepts timestamp
remove comment - apparently the code works fine when no servlets around
if a message goes to a vertex that doesn't exist, filter the message.
COMMERCE-<I> Use LocalService rather than Persistence
Removed extraneous logger calls
[refactor][checkpoint] Remove unused method in AlternatingController
Allow goog.scopes as part of .i.js processing\n\nThis is easy to support, and doesn't seem to be a significant slowdown for\ngenerating .i.js files.\n\n-------------\nCreated by MOE: <URL>
KristianRosenvold: Issue <I> fix\n\nr<I>
Added in ability to programmatically set the text color.
Bugfix: empty or no filter should match all scan results?
found minor bug but it has to wait
Class does not need to be public.
Remove accidental "sleep 5 seconds" at start of every mm-test test (#<I>)
fixes client security issue, now properly wrapping callable in SecureCallable
GUACAMOLE-<I>: Handle sublasses of Error in addition to RuntimeException.
minor refactor: a few utility methods
Fix usage of @class in SQL projections for embedded documents
Fix typo in comment in TransactionConfidence.java
Updating comment to be more clear
Made DefaultIndexRedirect a concrete class so it doesn't need to be subclassed.
Update Flickr4Java/src/com/flickr4java/flickr/REST.java\n\nAdded no-auth handling to POST too, so that flickr.test.echo can be called without signing the request.
remove extraneous line seen during testrun
make handle_start and handle_resume blocks get called on first load.
corrected url for userImpl.
[java] Renaming a test setup method to prevent overriding a method with the same name in the base class
Clear out our speak service once we've unregistered it.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/narya/trunk@<I> <I>f4-<I>e9-<I>-aa3c-eee0fc<I>fb1
Java Client: Making extracted methods visible for integration tests\n\n\nr<I>
Return InvalidRequest when using a keyspace that doesn't exist\npatch by jbellis; reviewed by pcannon for CASSANDRA-<I>
[DROOLS-<I>] allow to create a valid KieClasspathContainer even from a non-maven project
change `lockPostingChangeNotifications` variable from `protected` to `private` based on [ the code review feedback](<URL>)
Correct formatting\n\nThe new compare statement is now on a line of its own
JBPM-<I> - Case file conditional event fails other cases (#<I>)
Logging exceptions that occur in getMetrics API
Don't need those setters too, it's taken care of by the attributes.
test(metrics-es): refactor deprecated TestContainers methods
Listeners are now properly notified.
dont need to throw runtime exception in update
Added javadoc in CacheCreateConfigOperation
Moved the connection closer to the try block.
Deprecate Closeables.closeQuietly.\n-------------\nCreated by MOE: <URL>
UNDERTOW-<I> Make sure session is only cleared if it is the current session
Add Math.atan2() approximation
Inlined a no-longer-necessary local variable.
If the ejb-timers DB persistence is active non persistent timers should run without checking the DB as there is no entry. Otherwise such timers are never executed
Updated Misc convertToHex to use StringBuilder
Return overlapped nodes in-order
Removed tests that don't work due to inlining of callTyped()
A work around Javac closing environment resources after annotation processing phase, even if type annotations are present\n[Not ready for integration with langtools main branch yet]
Config settings do not use `--` in their names
Added Bi-methods to Throwing: BiConsumer, BiFunction, BiPredicate.
Covert the resource path to a package path
Issue <I>: Scan progress dialog can cause UI freezes
Pass through our parameters so that looping works just like playing and we can set positions and gain and such.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/nenya/trunk@<I> ed5b<I>cb-e<I>-<I>-a<I>-f6a<I>f<I>b<I>
no longer needed this reference to publications.json
Added Stratosphere Header to PlanConfiguration
Quote thread name in thread dumps\n\nA JVM thread dump surrounds the thread names with double-quotes\nand IntelliJ requires this to parse a thread dump.
fix satyan/sugar#<I>\n\nSometimes Throwable doesn't have a message,\nso e.getMessage is NULL and generate a NullPointerException
turn the integer division in the cleanup-interval into a bit mask and up the value by a large margin, although it does not really help the performance by any measurable degree.
Support JDK <I> in annotation processors (#<I>)
GLM coefficient bounds fix .
DefaultClientAdaptorImpl: fixed handling <I> error within importData method. If the method receives <I>, it doesn't retry.
UP-<I> Update to use LrsStatement
removed redundant final from named subresource setting builder
[JBAS-<I>] Create the DC's tmp deployment upload dir
Add facility to deserialize an instance of ErrorMessage from JSON
Add ability to set grace period on TemporaryJobBuilder
Control window and current product defaulting to null
disable debug\n\nthat was enabled by mistake in last commit
PUB-<I>: Fix deepfeatures() scoring (data transformation), was adding response column to\ndata transformation, because it thought the Autoencoder model was supervised.
fixing issues in the tests after the screenshots orientation fix.
Added sudo consumer for simplify update data in database from admin
[WFCORE-<I>] Make CredentialReference thread safe
OK, web cloud tests use both MISSING and FOUND as bug types... sigh.... Accomodate both\n\ngit-svn-id: <URL>
Fixed NullPointerException raised in isUserDefinedField
Improved indexing of modelgroups: indexed for all local-available.
[PAXEXAM-<I>] do not exclude required JARs from WAR probe
[Java]: Consistent ordering in code style.
Create constructor that takes all properties as parameters
[Java]: make sure to cast termLength to long so that 1GB term lengths compute correct log length.
FS.get(): Add URI causing exception to message.
Ignore test since it cannot be run concurrently with others until #<I> is fixed.
For Firefox we need to set parent and file detector explicitly the elements are created directly from hashmaps and not first converted into a remote element which is fed through converter again (like for chrome)
Throw MissingResourceException if constants lookup failed
FlatProperties: keep order of properties
Avoid logging a full stacktrace for expected errors, print it only at debug logging level
Using the root package as default for the classpath scan.
SONAR-<I> fix unstability of UT of api/users/create
[CONJ-<I>] correction for LRUTraceCache excption loggin
pubsub: Add removal comment in ConfigureForm
Disallowed use of names 'Utils' and 'Records' in the VDM model being generated as these are reserved by the Java code generator
Performance improvement - don't issue a log.trace with a huge string content concatnation, without first checking if trace is enabled
Improve creation of child path. This change is needed to allow the\nmerge logic to work on GCS.
Using callable statements against older server versions (<I> or <I>)\ndid not work because of recent changes to output parameter handling.\n<I> and <I> do not support function output parameters, but they\nstill must support the functions return value.  Every function must\nhave either a return value or an output parameter so we force the\nminimum output parameter count to one.
Ensure main thread is not interrupted when running testKillBeforeStart test. (#<I>)\n\nWhen the main thread is interrupted subsequent tests fail due to an IO channel being closed by the generated InterruptedException in that test.
If you're going to quote, do it right.
Overriding fillinstacktrace at alex's suggestion.
Switching VANILLA mode off, since too many changes in <I> repo and Oniguruma original (e.g JRUBY-<I>).\n\ngit-svn-id: <URL>
[Huobi] Return only opened orders when adapting orders
fix JavaDoc of callSuper in the ToString annotation
CORE-<I> Fixed missing DEPLOYMENT_ID column\n\n(cherry picked from commit <I>ad<I>d<I>f<I>f6f<I>b<I>a<I>d<I>bc<I>)
Up arrow key repeats the same command after pressing enter rather than going to the previous command (#<I>)
make the ServerEventTest not go out and do DNS lookup.
XRENDERING-<I>: ParserUtils#removeTopLevelParagraph keep the paragraph as children parent block
Test now reports when a file was not found in the classpath, instead of just throwing a NPE\n\ngit-svn-id: <URL>
fixed bug: wrong control flow in SessionSummary class
Performance improvement: instead of using Math.floor to detect precision loss, use Math.round (saves a Math.floor operation because in the happy path we'll be doing Math.round anyway)
fixed problem where errors and form data was being lost when a validation error occured during updating (ie following the edit view)\n\n\ngit-svn-id: <URL>
WSE-<I> - improved support for linux
PS-<I> renamed Probe to Sensor to better match the scalability rules naming
Keep "All Requests" option enabled state in sync with "HTTP Headers"\n\nChange method OptionsVariantPanel.initParam(ScannerParam) to add missing\ninitialisation of the enabled state of "All Requests" option, to keep it\ndisabled when the option "HTTP Headers" is not selected and enabled when\nit is.
for thread safe, use ConcurrentHashMap on ChannelThreadExecutorGroup
dah. sorry. last minute change without mvn clean insatll -- AND should be OR
New issue\nSummary: Remove "Service" from the default service-name in the new-service-wizard\nSince various service-components will be named as ${serviceName}-service it is a bad practice using "service" as a part of the serviceName resulting in service names containing "service" twice.\nE.g. entering the service name "myService" will result in a Mule service named "myService-service"...\nChanged the default name to "mySample" simply...
marked the http* methods from the servlet\napi as deprecated. they will be removed\nat some point...
attempt to fix slf4j-migrator Maven Central upload issue
return resolveAppName but wrap it in try/catch to be able to fallback
ENG-<I>: When a replica node rejoins or recovers, it is assumed that the cluster has already received the sync snapshot. Therefore we won't fail the rejoined node if another node goes down. Instead we assume that the prior instance of the cluster completed the snapshot processing. This prevents us from killing the rejoined node before it has connected to the producer cluster but after another node leaves.
Add ability to get default health check registry without an exception (#<I>)\n\nCurrently, there exists only one method for getting the default health\n\ncheck registry - `getDefault`. Unfortunately, it throws an\n\n`IllegalStateException` if the registry is not set. End users need to\n\ncheck for this condition. Some users would like avoid using\n\nexceptions for the control flow and would prefer to make a null check\n\ninstead catching an exception. For them we provide the `tryGetDefault`\n\nmethod which does exactly that.
JDateTime hashcode needs to use time, too.
Project health: accept other variations of nullability annotations.\n\nFor example NonNull(Decl|Type) and Nullable(Decl|Type) from the Checker Framework.\n\n<URL>
Make CollectionSubject.create() package private and cleanup the only caller.\n-------------\nCreated by MOE: <URL>
Rename toStringEquals to strEquals
[WFCORE-<I>] Correct the mechanism name comparison for HTTP Basic authentication.
TestWithStun: missing javadoc added
Actually updating capabilities with fixed noProxy
Using Guava to propagate exception instead of ifs
Testing colon escaping. Closes #4.
Add additional test case, 'assertArrayTypeWithNullIsNullSafe()' to ArrayPropertyUnitTests.
rename test only if in cluster mode
[FLINK-<I>][sql-client] Fix CliTableauResultView print results after cancel in STREAMING mode (#<I>)
prepare code for WFCORE-<I>
For ENG-<I>, add missing file
Handle Rhino IllegalStateException as parse error
Tweaking the JavaDoc a little\nMotivation:\nWe are now going to support two testing styles.  The Javadoc of the finder now explicitly states that it checks the old style of tests run by the instrumentation test runner.
Expose the resource accessor as a protected method.
removed redundant final from resource config applier
Charset by name --> StandardCharsets
Added a throw on impossible state
feat(Compare and Comply): Add attributes property to BodyCells model
Extended JBake watch mode to template and asset folder
Logging Tests Reset Correctly\n\nReset the logging results correctly; go back to the debug level as set in the gradle file
Improve robustness for unpacking so source\n\nReviewed By: lemon<I>\n\nDifferential Revision: D<I>\n\nfbshipit-source-id: d<I>c<I>f6d6ca<I>ff<I>b2c<I>a9eaf1a8a0fe
[Java] Fix wrap condition on active transport connection liveness test and only update on change to avoid cache invalidation.
Remove unexpected `MucNotJoinedException` from `MultiUserChat.leave()`\n\nOccupant information should be reset after `leavePresence`\nand `reflectedLeavePresenceFilter` are built.
* fix getExpandedItemsCount in the new StateManagment
Fix logging style in PinListSync
JGRP-<I> S3_PING.generatePreSignedUrl() has to use https for the\nprotocol\n\nI love a one-character fix!
java: Fixed RawType's bug that hash values of same byte arrays (raw data) are different
Infinite recursion (#<I>)\n\nMethod 'onClose()' recurses infinitely, and can only end by throwing an exception
Improve one coding style\n\n### What changes are proposed in this pull request?\n\nImprove one coding style\n\n### Why are the changes needed?\n\n### Does this PR introduce any user facing changes?\nNo\n\npr-link: Alluxio/alluxio#<I>\nchange-id: cid-6cefc7d<I>d8c<I>af9f<I>b5f<I>
Commit for io.openliberty.microprofile.lra<I>.internal\n\nThe existing try/catch will suffice for now. There are no situations where the select() will fail but get() will not\n\nIain has said this area will\nbe reworked shortly.
- NPE when canceling node instance that was dynamically added
ngrams: link the wiki page with instructions
Added comment with regards to replacement characteR
Throw a new IllegalStateException\n\nThrow a new IllegalStateException rather than the IllegalStateException returned from the cause\nof the ExecutionException.
Casting "n" to Number instead of Double for count command, since the server is inconsistent in what it returns (mongos returns Int<I>, mongod Double)
ENG-<I>: Deprecation warning if user tries to compile with project file.
Fixed lock checking bug in locale group service.
remove unused import (which fails on jdk<I>)
* Print exception class name and message in fallback stack trace printing.
fix "All shards failed"
[FLINK-<I>][Connector][Elasticsearch] Disable Elasticsearch disk allocation decider. Elasticsearch by default goes into read-only mode when the disk usage is above <I>%. On the Azure E2E tests, this is quite common.\n\nWe therefore disable the entire disk allocation decider, as documented on <URL>
Speed up GC of ByteSelectiveReader by clearing member variables in close()
Fix null name for middleware in Kotlin
Added Java .properties escape and unescape operations (for values)
Ensures FrozenPartitionTableTest waits for partitions update\n\nBefore sending partition operation, ensure\npartition table is updated on all members
#<I>: Added comment
Use class loader of parser base class in ClassWriter.\n\nEnsures that ClassWriter.getCommonSuperClass uses\nthe class loader of the parser class and not\nthe class loader of the ASM classes (which is the default).\nThis fixes class loading problems in OSGi environments.
We can't rely on the connection manager not being shutdown to determine\nwhether or not we're already shutdown as it may have shutdown\nunexpectedly.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/narya/trunk@<I> <I>f4-<I>e9-<I>-aa3c-eee0fc<I>fb1
EXOJCR-<I>: Fix LockJDBCConnection.
There's no benefit to surface being non-final in here.
removed some annoying out.println statements that shouldn't be there anyway
minor fix to tisana4j head method
ECKey.decodeFromDER(): Also throws IllegalArgumentException if unexpected class was read.\n\nFixes a regression that was introduced with commit e5f<I>e<I>a<I>bb1ed<I>e<I>a<I>eed<I>.
Fixed THORN-<I>\nmissing roles part for security pattern should result in "permitt all" and not in "deny all"
Fix for interpreter execution of properties #<I>\n\nFor some unknown reasons, the interpreter was not previously executing\nproperties and just always returned true for them.  This makes no sense and was\ncausing problems for quick check!
Attempt to fix issue #<I>
preExecute needs to be able to tweak PlexusConfiguration before its values are set to Mojo. Looks like <I> messed this up.\n\n\ngit-svn-id: <URL>
Prevent NPE from change in previous commit
C# ref. commit: Hmm, trimming away non-newlines seems to be a bad policy...
bug none: test uninitialised primitive fields
#<I> Copy of the branch
--print option with multiple input files would only print one file.
Unset CURRENT_REPEAT_ACTION after each phase
Log a briefer description of the place manager when it's unmapped.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/narya/trunk@<I> <I>f4-<I>e9-<I>-aa3c-eee0fc<I>fb1
Use correct range for period betwen calculation
Fix NPE in registeredLookup extractionFn when "optimize" is not provided. (#<I>)
[AS7-<I>] allow the server to be stopped first\n\nwas: d<I>d2bb9d1c0ab<I>ef2a3e4b<I>f5b2cae
Remove extraneous `else`.
Added cames-case text utilities.
[LO extension] fix bug: dynamic LT menus doesn't work
KeyChainGroup: defensive copy of the chains list in the c'tor
LegacyAddress: Update outdated JavaDoc reference.\n\nIt suggests using a deprecated method.
Updates JavaLangThrowable reference to NSException.
Update _UserResource.java
fix: fixed null pointer exception.
Added toString to BootChargingStationCommand
made JsfHelper.isFileFound tolerable towards none-file-based datastores
[java] Deleting unnecessary check, looping over an empty collection is OK
GWT-<I>: add getter for toolstrip.
reverted back to the previous version.
[#<I>][#<I>] only print "Unable to mkdirs" if dir did not exist && creation failed
synch no longer depends on davutil
Update DirectLogFetcherTest.java
Print a message when the JVM exits to help in debugging
Set CollapsingTitleBar background only once\n\nThis is a quick hack to address issue where the expended color was\nset after pop even if screen was collapsed.
rdma: reg_mr print failure reason
fix JsonProperty from activity_type to type
Temp use toString methods for "emitting" until "what to emit" is\nresolved.
Test case with a replicated table is on the outer side of an outer join with a partitioned table - no optimization
Allow access to SequenceNumber
Demo fix: allow https in URL field
Make CanvasTestBase abstract\n\nAlthough this doesn’t get run as a test under maven when using IntelliJ it attempts to run this class as a test which then fails, marking it as abstract means it should never get run by IntelliJ, but also allows it to continue to be used as before.
ATLAS-<I> JavaDoc NotificationConsumer and NotificationInterface - additional patch
fixed crash when using turkish locale
2.x: Fix typo in DisposableHelper (#<I>)
fix: javadoc for Streams
Bootstrapping JPA before the Goodies.
Reconnecting ObservableExecutionHook.onCreate\n\nSomehow this got lost in the refactorings over the past few weeks.
Incorporated Spencer's comment about shuffling oob rows.
Replace insane implementation with one that does not rely on the database to magically know in which order we want results returned.\n\ngit-svn-id: <URL>
Defined findClass method in class loader
Make icon enrichment a bit less verbose
added -jmx to help
Added command line expression for platform, and documented it.
Fixing wrong mapping of StaleElementReferenceException to error code <I>.\n\nIt is supposed to be <I>.
[OPENENGSB-<I>] removed unneeded code duplication
Fix issue with max age of v0 cookies
Move constructor as first method after field definitions
stabilize test: it should not depend on external browser links
fixed a bug related to text field width using the new layout
Suppress compiler warnings that aren't applicable in this case
Correct mismatched comments (#<I>)
#<I>  - allow to find more than one activejdbc.properties file on classpath and load them all
Fix to use Local interface instead of Remote
Updated the Amazon Service Catalog Service for Unprovision Product Action
#<I>: Add missing properties to 'build' factory methods.
- bugFix: Rest did not work runing efaps with jetty inside eclipse\n\ngit-svn-id: <URL>
Makes Kafka client refresh configuration on group id changes.
Ensure lazyPatternBy has same toString result as nested By
fixed the logo runtimeproperty
Fixes #8: Fix bug in client serializability.
- be sure only to use positive longs as random ID
Ignore this old test; this convention is not used anymore, AFAIK\n\nChange-Id: I<I>ea<I>a<I>cce<I>f<I>aac1c4e<I>ebabcaa2c
Improved and completed ListItems.
Make this benchmark more challenging (as if I needed that ;-))\n\ngit-svn-id: <URL>
Fix imports for ChannelSecurityConfigurerTests\n\ngh-<I>
Calculate HGVS only for SNVs
allow the m2 dir to be overwritten as we use a non standard location on the CD infra so we can mount it on a PV
Remove unnecessary S3 API call\n\nWhen creating a Resource from an ObjectSummary, we already know that the\nobject exists. Otherwise, we would not have an ObjectSummary about it.\n\nSo we can remove the exists() check, which causes a HEAD request for\nevery single matching object.\n\nThis reduces the number of HTTP calls from n<I> to 1.
Add more comment to the JoinProducer.
@since forgotten in b2cd<I>.
[Java] Remove -1 check from return of delay generator.
Change to parse unsinged long
DAT-<I> Case sensitive objects not being correctly handled in dependency management
Fixed surrogate order: high then low
[FIX] Rendering problems in Android Studio
Fix #<I> by removing the getContent that loses line endings because it doesn't seem bo be used by anywhere.
Modify TSMeta.incrementAndGetCounter() to pass new TSMetas through tree builders
Javadoc fix for DeploymentBuilder
Unit tests for SqlQueries expectations builder
some dead code copied from TinkerPop, a //TODO for someday
[findbugs] List bad == comparison
an exception thrown will be captured by both thenAccept and whenComplete\n\nwe remove one redundant logging here.
[Java] Error formatting.
SimonStewart: Weeding out a couple of unneeded calls to setSelected and toggle\n\nr<I>
Test the Gradle Plugin against Gradle <I>\n\nSee gh-<I>
Updated repository root RDF to use standard vocabulary sd:endpoint to\npoint at the fcr:sparql service.
Prevent potential int overflow on devices with <I>MB available heap
Fix issue with loadUpdateData GH-<I>
Fixed issue for ROO-<I> where transient keyword was lost\n\nTransient was missing from the modifier set util.
Fix Sigar dependency exposed into plugins
Fix typo in UsingToStringOrdering javadocs.\n-------------\nCreated by MOE: <URL>
minor refactor: rename local var
reduce number of graphs to make the test pass with <I> postgres max\nnumber of connections
Fire unsubscribe completion for detaches as well
Disable TestHiveRecoverableGroupedExecution\n\nThis test is still flaky even if only enabling one test, which\nmeans it needs more investigation.
Update BitmapHunter.java\n\nFix grammar mistake in comment
Add org.jenetics.util.Predicates which contains some general purpose predicates like Not, And, Or, Nil.
add ability of use TransitionSet of Transitions with different duration
Index object: check for null in the columns/toString()\n\nThis is an odd situation, but it occured during testing and fixing another bug. Since it should hardly be any performance penalty, I want to be on the safe side here.
fix(fsm): improve logging for missing event handler
Unignore tests that aren't failing on stable-<I>
Went back to only draining when we detect that the line is active.\n\n\ngit-svn-id: svn+ssh://src.earth.threerings.net/narya/trunk@<I> <I>f4-<I>e9-<I>-aa3c-eee0fc<I>fb1
[jdk9] added debugging information for when we cannot determine the compiler
Fixed ofYearDay tests to include the last day of the year.
Reintroduce constructor to UserProfile to preserve backward compatibility. See #<I>
#<I>: Additional test case.
Fix stage test issue when using headless executor
WanRESTTest should not execute in parallel (#<I>)\n\nWanRESTTest should not execute in parallel\n\n\n\nSince it uses real network, members could try and form clusters. Since\n\nthe network config is different, members will fail to start since they\n\ncannot join each other.
[Java] Make flags API in ExclusiveBufferClaim and Header consistent.
Correct SuppressWarning from unchecked to rawtypes so that latest Java 7\nwill allow it to compile.
#<I> use lower case for value of full calendar event.rendering (#<I>)
Synchronize creation and use of thread local looper to avoid null pointer exceptions
- Modified toString() to show the name of the file
Changed report widget to not unnecessarily scroll down for empty report updates.
throw Exception if FlagEncoder is defined twice
Convert '\\' to '/' when generating tree paths.\n\nDirectoryScanner produces paths with '\\' separators on\nWindows which need to be converted to '/' when used\nas Git tree paths.\n\nFixes issue #1
Return cached value under concurrent requests\n\nIn ClientDelegatingFuture when the current thread\ndoesn't win the CAS to set the cached deserialized\nvalue, return the same deserialized instance that\nis cached. The same behaviour applies in superclass\nDelegatingCompletableFuture.
fix Describe.shortly for SelenideElement: let it delegate to the original WebElement
JBRULES-<I>: setter for binary representation of ksession
Fix CPU time and Scheduled time in PlanPrinter
added private constructor to override default public constructor
[GOBBLIN-<I>] Improve MySQL staging table to have properties from dest table\n\nCloses #<I> from jinhyukchang/jnchang-jdbc
[bug <I>]: added ability to get URI for Path\n\nChange-Id: I<I>b<I>f<I>e<I>b8bdd<I>b<I>a3b7fb<I>ac4
bug fix. This needs to be kept across the doRun method.\n\ngit-svn-id: <URL>
[JENKINS-<I>] Don't use random for File name in Save to Workspace option (#<I>)
#<I>: Move class to base package.
ModelTest: printed missing property names were wrong
Use constant time equals for HashData
Don't map textual data sources
Migrate from deprecated `JCommander` constructor to calling `parse()` explicitly.\n\n(This also lets us remove an unnecessary `var unused` from the callsite.)\n\n#checkreturnvalue\n\nPiperOrigin-RevId: <I>
Trivial refactoring: simplify code
Add missing i<I>n method
Make sure that ElementType.TYPE_PARAMETER is only used on\ntype parameters.
Fix bug <I> - Codegen optimizer fails with Unsupported major.minor version <I> on JDK7
Add sender's address to warning on JoinRequest to be able to identify it.
[GOBBLIN-<I>] Fix NPE in DagManager cleanup\n\nUnder certain conditions, we can get a null list\nwhen checking if\ndag has a running job. This is a quickfix to\naddress that.\n\nCloses #<I> from aplex/dag-manager-npe
Removed shutdown hook from NodeService because it is already present in Services.
#<I> change LinkedIn Id type from int to string.
Default workers amount of RExecutorService set to 1
[ci] Fix spelling and display issue with javadoc
Added a mechanism to upgrade the channel to unrestricted one for\nprogrammatic clients.
[JBJCA-<I>] rar-info: Display transaction support in summary
fixing JENKINS-<I> (#<I>)
Fixed bug where we weren't closing the audio connection when the main websocket was closed. Impacted the reconnect system and possibly left phantom threads open.
Ensure buffer is flipped on read in InstallRequest.
JoinMatcher: looks like it works! Sort of...\n\nHad to implement .accept() (duh). However the Join rule does not appear in the\nparse tree...
Show more source after an exact error location.
Move Engine from Map 3 to Map 2
Apply non-TupleDomain predicates to Hive partition list\n\nWhen Hive connectors look up all partitions that is potentially needed,\nit didn't taken advantage of the additional predicate available in\nConstraint to apply the predicates that are not representable as\nTupleDomain.\n\nWhen Hive was migrated to the current TableLayout API, this additional\nfilter was not added. This is most likely caused by not thinking\nthoroughly enough and trying to retain whatever the code did with the\nold API.
[lakebtc] incorrect hostname capitalization results in SNI failure handshaking with server
demo: add commit to finish the setupOnApplicationOnCreate flow on demo application
Maintaining refresh behavior when there isn't a ServiceMatcher
Suppress all kinds of Exceptions raised by report() (#<I>)
- bug #<I>: the column reported by LanguageTool\n  was sometimes wrong when error span the new line.
make isIdElement() also here
Don't expect special names likes __iterator__ to be @private\n-------------\nCreated by MOE: <URL>
Change constructor visibility for WebSocketSubscribtionTransport (#<I>)
This concept is unique to View, so I think it's fair to implement this by default
Code Fix for SR id: EMBPD<I>\n\nRenamed from error to cancel status message
Do not recreate adapter if it does already exist.
Changed log on script.sleep.
adds Hostidentifier to better support uuids
ENG-<I>: fix handling of extra escape characters
Structure object unmarked as transient for serialization
fallback to double for timestamps
Removed unnecessary debuging info.
Activiti-Cycle: Now the "create default maven project" action should work with the SVN-Connector too.
Retuning all command names in alphabetical order
Replace null return with named constant to make intention clearer.
Re adds the CacheResult exception deprecating the mispelling and adding an exception with correct spelling.
Issue #<I> - Webview and Extensions option added to WebButton object
EXOJCR-<I>: some performance tweak
[Java] Log on level trace in DiscoveryEntryStoreInMemory
NCL-<I>: sped up GET /builds (by fixing querying for count)
Removing left over clear() method.
[Java] Tidy up.
fixes small bug introduced by CASSANDRA-<I>
turn off incomplete gbm test
GeoJsonObject: Added optional property "name" used by QGIS.
[Java] Group cycle threshold value with other time based values.
fixes #<I>\nlogged rejected execution exception while cleaning resources of a closing connection instead of throwing
AS7-<I>: Fix trailing space that is causing the build to fail\n\nwas: 4f<I>bdb0b<I>e2f<I>cb<I>f<I>f<I>e<I>
Removed a duplicate call to TaskStateTracker.registerNewTask()
A work around Javac closing environment resources after annotation processing phase, even if type annotations are present\n[Not ready for integration with langtools main branch yet]
Validate if name property is present in the response to avoid an NPE\n\nApparently, GitHub users who did not set a name will not have the\nname property present in the authenticated users response.
recycle buf for highspeed I/O
down to <I> test problems
Issue #<I>: Added test to FluxCircuitBreakerTest to verify that an error is propagated when you subscribe to Flux.error, but the error is not thrown onSubscribe.
Fix artifact extension retrieval\n\nDo not rely on the packaging type to figure out what the extension of the\nmain artifact will be. So far, using `jar` and `war` packaging for `.jar`\nand `.war` files worked by chance.\n\nWe know retrieve the actual extension as provided by Maven's\n`ArtifactHandler`.\n\nFixes gh-<I>
Closing DBCursor in finally block
Mark J2ObjCIncompatible as @Documented so that it shows up in the rendered Javadoc for Guava.
Add constructor that does not require id.
Use the same floating point math as XGBoost in objective functions
[CONJ-<I>]  ssl cipher test correction
[de] add suggestion for "unzwar"
Update DataIO.java\n\nThe method's (unpackInt(DataInput is) ) parameter name is a naming convention smell. So it might be renamed as unpackInt(DataInput in).
The VelocityEngine export is no longer optional
Leave 2 threads for the operating system.
Expose slowInvocationThresholdInMillis and verySlowInvocationThresholdInMilis as writable JMX attributes
ENG-<I>: Eliminate race caused by stats agent being initialized too early.\n\nEnable the stats agent after all system components are constructed to\navoid stats request handling racing with the construction of the\nsub-systems.
Updated fingerprinter code to get paths from an atom at one go, rather than multiple calls to getPathsOfLength for each path length. Gives a nice speedup\n\ngit-svn-id: <URL>
Handled exception when no ManProvider has been defined
Issue #<I>: Sync should support non zip archives for Android
[refactoring] changed getAllExtensions to getExtensions
get String s in variable for the method gotToMail
HibernateBundle requires at leaat one entity class.
Allow setting object ACL during PUT\n\nReferences #<I>.
bug/#<I>_2 - removed redundant call to scroll events\n\nImpacted class:\n* `MapController`: removed redundant call to scroll events from method `setCenter` (that already calls `MapView.setExpectedCenter`)
[xbase][contentAssist] Added more CA tests + some fixes
WYIL: tweaks for bugs caused by the switch away from using Assign.
fixes android NPE by changing java template
Transaction: Print transaction version to toString().
Removing previously added System.out.prinltn's
ARQ-<I> Proxy server side exceptions and re-create them on the client if possible  (removed debug material)
Fixes a minor bug that results in more forward declarations than necessary.
Warm up partitions before starting tests\n\nWhen the partitions are not warmed up and the test starts, the partition assignment will start. In some cases the partition assignment lasts for a long time, causing invocations to fail on some members.\n\nFixes failing assertions (not timeouts) in <URL>
Give singleton a default of true.
[JENKINS-<I>]\nwe are looking for dependencies since the last successful build\nand not since the last previous not failed build
test fixing hudson issue
[FLINK-<I>][metrics] Schedule latency markers with fixed delay
Selector optimization logging noise fix\n\nInstead of logging on info if the selector got optimized, it is now\nlogged on finest. We don't want to see the noise every time HZ\nstarts.
Remove dummy exception from async test util
Set alt attribute of current image in DropDownImageListEditor for selenium
Check only data nodes in Elasticsearch disk usage stats (#<I>)
typeArgumentsForType should use resolved types
Pass --provision to vagrant up to force provisioning\n\n- sometimes vagrant seems to be confused about whether its a first boot or not\n  and does not provision the created VM
override emit(Notification)\n\nto delegate the notification emission to the primary context
CDK-<I>: Fix avro schema comparison bug.\n\nThis adds a check that the number of sub-fields in a record matches.\nOtherwise, when fields are added at the end of the record, equals may\nreturn incorrectly depending on which schema is the left operand.
- adding more logging output in case REST request fails
Fixed an issue where the parser was returning an Exception when it shouldn't have been. Removed some unneccesary null checks.
[FLINK-<I>][py] Check for null when closing socket
Removed usageCount - why we would need one :)
[polish] Fix MaterializeSubscriber toString, AbstractQueue's won't work
Address javadoc warnings in fcrepo-kernel-api (#<I>)\n\n- include add '@throws IOException' comment for init() method\n\n\n\nResolves: <URL>
Now logging stacktrace of exception when instanciation fails
Modifications and development for the issue\n"First time user registration - Collect E-Mail from user on instance activation" #<I>
removed unneed constructor from PUBREC packet
Added a test for a large date format #<I>
Remove a print statement that I accidentally left in a test.\n-------------\nCreated by MOE: <URL>
Always run the stand-alone app in production mode.
#<I>: Updated documentation for ProfileType to reflect changes made in issue #<I>.
GRPC: recognise appSecurity-<I>\n\nAdd appSecurity-<I> to grpc's list of security features
Add support for different UTC timezone id used by IBM JVMs
Tune NUM_FILES_PER_CHUNK based on timed tests
[language-translation] Allow more than one file to be used when\ncreating a model, fix #<I>
Notify arguments are only returned when using the V3 protocol, so\nskip the notify argument test on a V2 connection.
Fixed an embarassing bug where the wrong rejected notification listener would be removed if a weak reference expired.
Print some useful information about feature assumptions during test runs.
fix spotless (getsentry/sentry-android#<I>)
For ENG-<I>, cleanup based on Ning's feedback
ECDDEV-<I> Updating the model class
Fix second 'created' to read 'edited'.
Adjusted the ordering of calls to ThreadPoolExecutor.setMaximumPoolSize\nand setCorePoolSize to avoid the case where we can potentially bump the\ncore pool size beyond the max and THEN adjust the max.
Add generic specialization and remove unused field in FxmlLoadResult
app: last chunk is CADD score parser is now also serialized
Add method to internal builder that builds while excluding some Java source files. (#<I>)
WFLY-<I> make it easier for GC to clean up references
Improvement on JsArrayList#retainAll.
(#<I>) Added some comments to test
Fix for Bug <I> � Continuation support missing in Rhino <I> (cvs HEAD)
[Java] Record candidate term id to mark file even when a single node cluster to be consistent.
add a sys property "findbugs.noSummary" which suppresses the generation of the summary report in the gui version. On a sample project that takes 3M to find bugs, it uses over 7M extra to generate the summary, before getting out of memory errors. By using this option, large projects can still be processed.\n\n\ngit-svn-id: <URL>
Remove unused field userId in OrganizationDto
Don't wait for load if the url is a fragment for current url
Added whitespaces as requested by @raphaelbauer.
* Remove unnecesary check around toString()
jdk<I> switches private class calls from INVOKE_SPECIAL to INVOKE_VIRTUAL, so handle private INVOKE_VIRTUAL to A0 calls as well for #<I>
- removed unused imports.
fix post-ByteBuffer hint delivery.  patch by Sylvain Lebresne; reviewed by jbellis for CASSANDRA-<I>\n\ngit-svn-id: <URL>
Fixed image progress to work as specified.
SONAR-<I> Update the path of existing resources during migration
Issue #<I> File limit was changed to <I>.
make sure the task execute in order and not missing the chance for execution.
Remove excess code, no hava long.class in ShardingPropertiesConstant enum. (#<I>)
Fix a comment in class AndPermission.
Better defect localization in PidRuntimeFactoryTest
minor fix to rename edit log
Update Fake.class to support OpenJ9 Field Layout\n\nCurrently lombok crashes when building with OpenJ9 JDK as described in #<I>\n\nIt is due to the `override` field offset from Fake.class doesn't match actual offset value for AccessibleObject.class in OpenJ9 \n\nThis change is to support the OpenJ9 Field layout algorithm.
ENG-<I>: It is possible for an export message to be smaller than 8 bytes, in which case when it is truncated, the required buffer will be larger than original data because of the 8 byte header that is prepended to each buffer. This fix addresses this rare condition by ensuring that the allocated buffer accounts for the extra 8 bytes needed at the beginning of the buffer.
Fix formatting issue that fails mvn:validate
Removed spurious System.out.println
Stop exchange operator immediately when task output is done
added long click listener to treeview and folder structure example
Throw BugOrBroken if a ValidationFailed is created with no problems\n\nReported by Max in blog comments, thanks.
Delete temp file when integration test passed
removed unused import in sub resource setting
forgot to bind PriorityResource
Now only setting the ThreadLocal values when they have changed.
fix: reset token counter on pattern start
Make the mp workload less bursty
CORE-<I> DB2: custom-generated indexes for primary keys are not preserved in diff/generateChangeLog
[Java]: Better approach to tracking the index 2 ahead on publication roll over.
Removed the equalsPrime thing, because that's just needlessly putting implementation details in there. If switching primes is so important, hash the type name and turn that into a prime, or something.\n\nAlso added some javadoc.
Refuse snapshots during bootstrap.\nPatch by brandonwilliams, reviewed by thobbs for CASSANDRA-<I>
don't try to find tools.jar if compile is false
Don't construct a debug message unless that channel is enabled. Takes serialize() out of the profiles.
Added Javadoc for compareAndSet
#<I> [DB2 Support] - ModelTest.java - Prevent test failure in DB2
relax timing restrictions on PeriodicalRefreshStrategy test
Fix Firefox 2 launch on Ubuntu 7.x: Set library path when creating profile\n\nr<I>
actuall update user information in initialization
First bug fix relating to change in slot allocation.
Fix Swagger docs test for Business central (#<I>)
Use ms to track end time
Added unit test for multibyte problem
Fix a issue that inputting a string which is not numeric would cause a NumberFormat exception. (#<I>)
netty transport test: adjust timeout for some tests.
Update MPRestClient.java\n\ngetAccessToken condition is bad
Fix IntelliJ warning that iterator iterates over Object instead of String
[hotfix][task] Optimize and simplify Task#isBackPressured
ISPN-<I> GetAllCacheNotFoundResponseTest.createBeforeClass fails
fix issue: KVStore as sub class of Map cause failure in BeanSpec.of
Fixed PerformanceStateContainerTest for new performance output.
Start to restructure layer into\n\n1) per-run layer parameters\n2) per-thread layer state (activity, error)\n3) per-node shared model state (weights, biases and momenta)
Fix a bug and correctly check all proto init field types even in the presence of unknown field types.\n\nBefore this, the compiler would incorrectly stop verifying proto init field types after it encountered a field type of ? or list<?>.\n\nGITHUB_BREAKING_CHANGES=You may need to correct field types for proto init expressions where a field is type ? or list<?>.\n\n-------------\nCreated by MOE: <URL>
Add a get method for the policy rule URL.
feature/JENKINS-<I>-delivered-size-header * Add another response header to log requests to show delivered size in bytes for paging (#<I>)
Adds the Permissions function to the NSActivity (1/2 patches)
Project health: sync access to Thread.interrupted variable.\n\nPiperOrigin-RevId: <I>
When a DG fails due to port allocation issues, hint what the problem might be\n\nThis commit adds the note in the parentheses below:\n\n  "Allocation of port <port> for job <dg-job> collides with job <other-job>\n  on host <host> (the conflicting job was deployed manually or by a\n  different deployment group)"
[LO extension] solves issue #<I>
#<I> Add missing Deprecated annotations
Minor improvement in checking for whitespace in URIs
Merged if statement with enclosing one
